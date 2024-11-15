import sys
import numpy as np
class HMM:
    def __init__(self, a, b, c, d, f, evidence):
        """Initializes the Hidden Markov model with the a-f variables and the boolean evidence

        Args:
            a (float): X_0 (true) value 
            b (float): P(X_t | X_t-1)
            c (float): P(X_t | ¬X_t-1)
            d (float): P(e_t | X_t)
            f (float): P(e_t | ¬X_t)
            evidence (array): List of T or F evidence
        """
        # <P(X_t | X_t-1), P(¬X_t | X_t-1)> ,   <P(X_t | ¬X_t-1), P(¬X_t | ¬X_t-1)>
        self.transition_matrix = np.array([[b, 1 - b], [c, 1 - c]]) 

        # <P(e_t | X_t), P(¬e_t | X_t)> ,   <P(e_t | ¬X_t), P(¬e_t | ¬X_t)
        self.observation_matrix = np.array([[d, 1 - d], [f, 1 - f]])

        # <P(X_0), P(¬X_0)
        self.initial_state = np.array([a, 1 - a])  

        # Binary evidence
        self.evidence = evidence

    def compute_prob(self):
        """Computes the probability P (Xt|e1:t)

        Raises:
            ValueError: If there is no valid character for the evidence value

        Returns:
            state_probs (tuple): The normalized state probability
        """
        # Setting initial state
        state_probs = self.initial_state

        for e in self.evidence:
            # Prediction step: <Px,¬Px>
            predicted_state_probs = np.dot(state_probs, self.transition_matrix)

            # Update step based on current evidence
            if e == 't':
                obs_probs = self.observation_matrix[:, 0]  # Observation likelihood for e=true
            elif e == 'f':
                obs_probs = self.observation_matrix[:, 1]  # Observation likelihood for e=false
            else:
                raise ValueError(f"Invalid evidence value: {e}") # In case there is something that isnt t or f

            # Element-wise multiplication
            updated_state_probs = predicted_state_probs * obs_probs

            # Normalization
            normalization_factor = np.sum(updated_state_probs)
            state_probs = updated_state_probs / normalization_factor

        return tuple(state_probs)

class HMMProgram:
    def __init__(self, input_file):
        """Initializes the program to process the cpt.txt file and run the HMM computation

        Args:
            input_file (string): The file given in the sys argument
        """
        self.input_file = input_file
        self.parsed_data = [] # Storing the input file to use it later
    
    def process_file(self):
        """Opens and reads the file. Parses the data and maps it to the approviate variables
        """
        with open(self.input_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split(',')
            a, b, c, d, f = map(float, data[:5])
            evidence = data[5:]

            # Storing parsed data + original line for output formatting
            self.parsed_data.append((a, b, c, d, f, evidence, line.strip()))

    def run_HMM(self):
        """Runs the HMM algorithm by calling the HMM class and its compute_prob function.

        Returns:
            results(string): The formatted results to be printed out
        """
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