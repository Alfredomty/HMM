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
        self.evidence = evidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hmm.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]