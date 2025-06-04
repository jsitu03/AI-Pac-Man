from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        states = self.mdp.getStates()
        for iter in range(self.iters):
            current_values = {}
            for state in states:
                maxVal = float('-inf')
                actions = self.mdp.getPossibleActions(state)
                
                for action in actions:
                    currVal = self.getQValue(state, action)
                    maxVal = max(maxVal, currVal)
                    current_values[state] = maxVal
                    
            self.values = current_values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
    
    def getPolicy(self, state):
        
        if self.mdp.isTerminal(state):
            return None
        
        else:
            bestVal = float('-inf')
            bestAction = None
            for action in self.mdp.getPossibleActions(state):
                currQValue = self.getQValue(state, action)
                if currQValue > bestVal:
                    bestVal = currQValue
                    bestAction = action
                    
            return bestAction
        
    def getQValue(self, state, action):
        curr_values = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        for coord in curr_values:
            reward = self.mdp.getReward(state, action, coord[0])
            qValue = coord[1] * (reward + (self.discountRate * self.getValue(coord[0])))
            total += qValue
            
        return total