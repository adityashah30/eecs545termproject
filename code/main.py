from ga import GA

def main():
    pop_size, gen_count, mutation = 100, 100, 0.3
    solve_method = "logistic"
    print "Using solve method: ", solve_method
    full_accuracy = GA.full_accuracy(solve_method)
    print "Full Accuracy: ", full_accuracy.fitness
    print full_accuracy.feature_subset
    solver = GA(gen_count, pop_size, mutation, solve_method)
    finalPop = solver.search()
    print finalPop[0].fitness
    print finalPop[0].feature_subset

if __name__ == '__main__':
    main()
