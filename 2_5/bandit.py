class Bandit:
    def __init__(self, distributions):
        for d in distributions:
            if not(len(d) == 2 and isinstance(d[0], float) and isinstance(d[1], float)):
                raise ValueError("Each distribution must be a tuple of two floats")
                
        self.distributions = distributions