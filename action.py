class Action:
    def __init__(self, action_type):
        self.action_type = action_type
        assert(self.action_type in range(9))

    def get_action_type(self):
        return self.action_type
