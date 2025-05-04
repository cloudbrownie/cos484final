from tot.tasks.game24 import Game24Task


task = Game24Task()

print(task.test_output(986, "7 - 1 = 6 (left: 2 6 12)\n6 * 2 = 12 (left: 12 12)\n12 + 12 = 24 (left: 24)\nAnswer: ((7 - 1) * 2) + 12 = 24"))