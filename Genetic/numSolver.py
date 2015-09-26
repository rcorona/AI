""" 
Rodolfo Corona
numSolver.py

A genetic algorithm to find a solution to a given number. 

e.g. 2*22 is a solution to 44. (Operators are used sequentially from left to right,
in other words, precedence does not matter.)
"""

import random
import sys

#Encodes a solution to the problem. 
#NOTE: Genes numbered 1 - NUM_GENES. 
class Solution(object):

    def __init__(self, chromosome = None):
        if chromosome is None:
            self.chromosome = random.getrandbits(NUM_GENES * GENE_BITS)
        else:
            self.chromosome = chromosome

    def getGene(self, num):
        mask = (0xF << GENE_BITS * (NUM_GENES - num))

        maskedGene = (self.chromosome & mask)

        geneBin = (maskedGene >> GENE_BITS * (NUM_GENES - num))

        return encodings[geneBin]

    def getGeneString(self, num):
        return str(self.getGene(num))

    def getGenesString(self):
        string = ""

        for gene in range(1, NUM_GENES + 1):
            string += " " + self.getGeneString(gene)

        return string

    def printGene(self, num):
        print self.getGeneString(num)

    def printGenes(self):
        print self.getGenesString()

    def getOperator(self, geneNum, score):
        #Base case. 
        if (geneNum > NUM_GENES):
            return None, score

        val = self.getGene(geneNum)

        #Looks for next available operator gene. 
        while (geneNum < NUM_GENES and ((not type(val) is str) or val == 'N/A')):
            geneNum += 1
            val = self.getGene(geneNum)

        #No next operator available. 
        if ((not type(val) is str) or val == 'N/A'):
            return None, score

        return self.getNum(geneNum + 1, score, val)

    def getNum(self, geneNum, score, operator):
        #Base case. 
        if (geneNum > NUM_GENES):
            return None, score

        val = self.getGene(geneNum)

        #Looks for next available number gene. 
        while (geneNum < NUM_GENES and not type(val) is float):
            geneNum += 1
            val = self.getGene(geneNum)

        #No next number is available. 
        if (not type(val) is float):
            return None, score
 
        #Case for first encountered #.     
        if (score == None):
            newScore = val
        else:
            newScore = score

        #Updates score by using number found and current operator.
        if (not operator == None):
            newScore = self.updateScore(score, operator, val)

        return self.getOperator(geneNum + 1, newScore)

    def updateScore(self, score, operator, rval):
        if (operator == '+'):
            return score + rval
        elif (operator == '-'):
            return score - rval
        elif (operator == '*'):
            return score * rval
        elif (operator == '/'):
            if rval == 0.0:
                return float("inf")
            else:
                return score / rval

        #Should never occur. 
        sys.exit("Error: Operator invalid!") 

    def getScore(self):
        val, finalScore = self.getNum(1, None, None)

        if finalScore == None:
            return float("inf")

        return finalScore

    def getFitness(self):
        score = self.getScore()

        if (target == score):
            return float("inf")
        else:
            return 1.0 / abs(target - score)

    def mutate(self):
        for i in range(0, CHROMOSOME_BITS):
            if (prob(mutateRate)):
                self.chromosome ^= (1 << i)

        return self

#Normalizes a list of values.
def normalize(list):
    total = sum(list)

    return [(value / total) for value in list]

#Samples from a PDF
def sample(population, probabilities):
    val = random.uniform(0,1)
    member = 0
    total = probabilities[member]

    while total < val:
        member += 1
        total += probabilities[member]

    return population[member]

def prob(val):
    if random.uniform(0, 1) < val:    
        return True
    else:
        return False

def crossover(parent1, parent2):
    if prob(crossRate):
        crossBit = random.randint(0, CHROMOSOME_BITS)

        mask = 1

        #Creates mask for crossover. 
        for i in range(1, crossBit):
            mask |= (1 << i)

        notMask = ~mask

        #Produces two children by crossing over the chromosomes. 
        chrome1 = (parent1.chromosome & mask) | (parent2.chromosome & notMask)
        chrome2 = (parent1.chromosome & notMask) | (parent2.chromosome & mask)

        return Solution(chrome1), Solution(chrome2)
    else:
        return parent1, parent2


def getOffspring(population, probabilities):
    parent1 = sample(population, probabilities)
    parent2 = sample(population, probabilities)

    child1, child2 = crossover(parent1, parent2)

    return child1.mutate(), child2.mutate()

def searchForSolution(population):
    best = None
    minDiff = float("inf")

    for member in population:
        score = member.getScore()

        if score == target:
            return member, True
        elif abs(score - target) < minDiff:
            best = member

    return best, False

#Seeds random number generator with system time. 
random.seed()

#Number of genes per chromosome. 
NUM_GENES = 9

#Number of bits per gene. 
GENE_BITS = 4

#Num bits in chromosome.
CHROMOSOME_BITS = NUM_GENES * GENE_BITS

#Population number. 
N = 200

#Max number that may be reached with given number of genes. 
MAX_REACHABLE = 9 ** ((NUM_GENES / 2) + 1) 

#Crossover rate
crossRate = 0.7

mutateRate = 0.001

encodings = { 0: 0.0,
              1: 1.0,
              2: 2.0,
              3: 3.0,
              4: 4.0,
              5: 5.0,
              6: 6.0,
              7: 7.0,
              8: 8.0,
              9: 9.0,
              10: '+',
              11: '-',
              12: '*',
              13: '/',
              14: 'N/A',
              15: 'N/A',
              16: 'N/A'
            }


if len(sys.argv) >= 2:
    target = float(sys.argv[1])
else:
    target = random.randint(0, MAX_REACHABLE)


population = [Solution() for i in range(0, N)]
answer, found = searchForSolution(population)
generation = 1

while not found:
    print "Best so far: ", (answer.getScore(), answer.getGenesString())

    newPopulation = []
    fitnessList = [solution.getFitness() for solution in population]
    probabilities = normalize(fitnessList)

    for i in range(0, N / 2):
        child1, child2 = getOffspring(population, probabilities)
        
        newPopulation.append(child1)
        newPopulation.append(child2)

    population = newPopulation
    generation += 1
    answer, found = searchForSolution(population)

print "Found solution in generation %d: " % generation
answer.printGenes()
