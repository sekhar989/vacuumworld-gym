import vacuumworld
from vacuumworld.vwc import action, direction

class MyMind:

    def decide(self):
        pass #return actions
        
    def revise(self, observation, messages):
        pass #belief revision

vacuumworld.run(MyMind())