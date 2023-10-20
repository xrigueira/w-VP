
def voter(*args):
    total = sum(args)
    return total / len(args)

scores_Xs_high = 1.0
scores_Xs_med = [1.0, 1.0, 0.7, 0.1]
scores_Xs_low = [1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0]

multiresolution_vote = voter(scores_Xs_high, *scores_Xs_med, *scores_Xs_low)

print(multiresolution_vote)