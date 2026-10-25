"""Pizza kitchen for policies trained on yaditya/qual-cheese-salami-final-model.

That dataset labels every episode with both the topping and the target quadrant:
`pick-n-put-{cheese|salami} put-in-q{1..4}`. Works with SmolVLA checkpoints and with
ACT checkpoints from train_act_qual_colab.ipynb (task one-hot via task_vocab.json; the
kitchen refuses to start if any recipe task is missing from that vocabulary).

Runs on its own port and shares the pizza/bin calibration in --work with the other kitchens.
Only one kitchen (or follow.py) may drive the robot at a time.
"""

from kitchen import Kitchen, main

QUAL_TEMPLATE = "pick-n-put-{topping} put-in-q{quadrant}"

if __name__ == "__main__":
    main(Kitchen, __doc__, task_template=QUAL_TEMPLATE, port=8898)
