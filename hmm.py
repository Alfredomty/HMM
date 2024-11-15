import sys
import numpy as np
class HMM:
    def __init__(self, a, b, c, d, f, evidence):

        # X_t-1 True: b true, 1-b false 
        # X_t-1 False: c true, 1-c false
        self.transition_matrix = np.array([[b, 1 - b], [c, 1 - c]]) 

        # X_t True: d true, 1-d false
        # X_t False: f true, 1-f false
        self.observation_matrix = np.array([[d, 1 - d], [f, 1 - f]])

        # P(X0) = a true, 1-a false
        self.initial_state = np.array([a, 1 - a])  

        # Binary evidence
        self.evidence = evidence

    def compute_prob(self):

        # Setting initial state
        state_probs = self.initial_state

        for e in self.evidence:
            # Prediction step: <Px,-Px>
            predicted_state_probs = np.dot(state_probs, self.transition_matrix)

            # Update step based on current evidence
            if e == 't':
                obs_probs = self.observation_matrix[:, 0]  # Observation likelihood for E=true
            elif e == 'f':
                obs_probs = self.observation_matrix[:, 1]  # Observation likelihood for E=false
            else:
                raise ValueError(f"Invalid evidence value: {e}") # In case there is something that isnt t or f

            # Element-wise multiplication for the update step
            updated_state_probs = predicted_state_probs * obs_probs

            # Normalization
            normalization_factor = np.sum(updated_state_probs)
            state_probs = updated_state_probs / normalization_factor

        return tuple(state_probs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hmm.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]