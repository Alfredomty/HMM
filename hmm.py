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

class HMMProgram:
    def __init__(self, input_file):
        self.input_file = input_file
        self.parsed_data = [] # Storing the input file to read it
    
    def process_file(self):

        with open(self.input_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split(',')
            a, b, c, d, f = map(float, data[:5])
            evidence = data[5:]

            # Storing parsed data + original line for output formatting
            self.parsed_data.append((a, b, c, d, f, evidence, line.strip()))

    def run_HMM(self):
        if not self.parsed_data:
            self.process_file()  # Process the file if empty

        results = []
        for a, b, c, d, f, evidence, original_line in self.parsed_data:
            # Create an HMM instance, compute probabilities
            hmm = HMM(a, b, c, d, f, evidence)
            p_true, p_false = hmm.compute_prob()

            # Format result
            result = f"{original_line}--><{p_true:.4f},{p_false:.4f}>"
            results.append(result)

        return results

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python hmm.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]

    program = HMMProgram(input_file)

    # Run HMM 
    results = program.run_HMM()

    # Output the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()