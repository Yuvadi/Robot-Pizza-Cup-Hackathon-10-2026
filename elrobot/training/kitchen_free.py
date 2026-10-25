"""Free-placement pizza kitchen: test policies that pick a topping without a target quadrant.

For checkpoints trained on plain tasks such as `pick-n-put-cheese` / `pick-n-put-salami`
(e.g. toppings-smolvla-multitask-policy). The order still asks for pieces per quadrant;
after each piece the calibrated pizza view tells which quadrant changed most, so every
step reports where the piece actually landed and whether it hit the requested quadrant.
A step is retried only when nothing landed on the pizza (or, with a bin box, nothing was
picked); a piece that lands in the wrong quadrant is tallied, not re-placed.

Runs on its own port and shares the pizza/bin calibration in --work with kitchen.py.
Only one kitchen may cook at a time: both drive the same robot.
"""

from kitchen import QUADRANTS, Kitchen, change_score, crop_quadrant, main


class FreeKitchen(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tally = self.empty_tally()

    @staticmethod
    def empty_tally():
        return {"on_target": 0, "off_target": 0,
                "landed": {f"q{q}": {} for q in QUADRANTS}}

    def verifiable(self):
        return bool(self.layout.get("box"))

    def check(self, step, before, after) -> dict:
        cam = self.layout["camera"]
        changes = {q: change_score(crop_quadrant(before[cam], self.layout, q),
                                   crop_quadrant(after[cam], self.layout, q)) for q in QUADRANTS}
        best = max(changes, key=changes.get)
        landed = best if changes[best] >= self.args.verify_change else None
        result = {"passed": landed is not None, "landed": landed, "on_target": landed == step.quadrant,
                  "changes": {f"q{q}": round(v, 4) for q, v in changes.items()},
                  "min_change": self.args.verify_change}
        if self.args.verify_pick > 0 and step.source in self.layout["bins"]:
            pick = change_score(self.bin_crop(before, step.source), self.bin_crop(after, step.source))
            result["bin_change"] = round(pick, 4)
            result["picked"] = pick >= self.args.verify_pick
            result["passed"] = result["passed"] and result["picked"]
        if result["passed"]:
            self.tally["on_target" if result["on_target"] else "off_target"] += 1
            spot = self.tally["landed"][f"q{landed}"]
            spot[step.topping] = spot.get(step.topping, 0) + 1
        return result

    def start(self, order, confirm, mystery_slot="cheese"):
        super().start(order, confirm, mystery_slot)
        self.tally = self.empty_tally()
        self.note("Free placement: the policy has no quadrant input; landing quadrants are measured, not steered.")
        if not self.verifiable():
            self.note("WARNING: pizza area not calibrated; landing quadrants cannot be measured.")

    def state(self):
        return {**super().state(), "mode": "free", "tally": self.tally}


if __name__ == "__main__":
    main(FreeKitchen, __doc__, task_template="pick-n-put-{topping}", port=8896, no_verifier=True)
