# Hopfield Usage Pattern


from neupy import algorithms
from random import randint
import numpy as np
from random import randint
from numpy import linalg
from sklearn.preprocessing import normalize
import scipy.spatial.distance as distance
from sklearn.metrics import jaccard_similarity_score
from random import *
import matplotlib.pyplot as plt
from matplotlib import style

style.use ( 'ggplot' )


# import hopfield_nets


# map(chr, range(97, 123)) -- lower case
# map(chr, range(65, 91)) -- upper case


def generateDataSets():
    num_data = 25
    class_label = list ( map ( chr, range ( 65, 65 + num_data ) ) )

    return class_label


def train(patterns):
    r, c = patterns.shape
    # print('Training...' , r , c)

    # patterns = patterns.reshape((1 , c * c))

    dhnet = algorithms.DiscreteHopfieldNetwork ( mode='async', n_times=2500, check_limit=False )
    dhnet.train ( patterns )

    return dhnet


def predict(check_pattern, dh):
    r, c = check_pattern.shape
    # print('Checking data/....')
    check_pattern = check_pattern.reshape ( (1, c * c) )
    result = dh.predict ( check_pattern, n_times=2500 )

    return result


def createUsagePattern(usage_pattern, data_labels, train_links, train_str=''):
    # Relation rules
    # A - B 01
    # B - D 13
    # A - C 02

    len_tl = len ( train_links )

    for t in train_links:

        # if t == 'FLM':
        # print(t)

        if len ( t ) == 2:
            x, y = t
            i, j = data_labels.index ( x ), data_labels.index ( y )
            usage_pattern[i][j] = randint ( 150, 250 )
            usage_pattern[j][i] = usage_pattern[i][j]

            if train_str == 'base':
                usage_pattern[i][j] = 1
                usage_pattern[j][i] = usage_pattern[i][j]


        elif len ( t ) == 3:
            x, y, z = t
            i, j, k = data_labels.index ( x ), data_labels.index ( y ), data_labels.index ( z )

            # print(t , i , j , k)
            usage_pattern[i][j] = randint ( 150, 250 )
            usage_pattern[i][k] = randint ( 150, 250 )
            usage_pattern[j][k] = randint ( 150, 250 )

            usage_pattern[j][i] = usage_pattern[i][j]
            usage_pattern[k][i] = usage_pattern[i][k]
            usage_pattern[k][j] = usage_pattern[j][k]

            if train_str == 'base':
                usage_pattern[i][j] = 1
                usage_pattern[i][k] = 1
                usage_pattern[j][k] = 1

                usage_pattern[j][i] = usage_pattern[i][j]
                usage_pattern[k][i] = usage_pattern[i][k]
                usage_pattern[k][j] = usage_pattern[j][k]


        elif len ( t ) == 1:
            i = data_labels.index ( t )
            usage_pattern[i][i] = randint ( 50, 80 )
            if train_str == 'base':
                usage_pattern[i][i] = 0

    return usage_pattern


def calcCosineSimilarity(mem_pat, test_pat):
    r, c = mem_pat.shape

    mp = mem_pat.flatten ()
    tp = test_pat.flatten ()

    # print(mp)
    # print(tp)

    cos_sim = 1 - distance.cosine ( mp, tp )

    # print(jaccard_similarity_score(mp , tp))
    return cos_sim


def createTestPatterns(labels, n_cols, shift=2):
    link = 2

    test_rel = []

    epoch_step = 2

    for i in range ( 0, randint ( epoch_step - 1, len ( labels ) ) ):
        for j in range ( i + 1, len ( labels ) ):
            if j + 1 < len ( labels ):
                temp_str = labels[i] + labels[j] + labels[j + 1]
                test_rel.append ( temp_str )

    # print(test_rel)

    # test_pattern = createUsagePattern(test_pattern , labels , test_rel)

    return test_rel


def normalize_mat(pattern):
    coeff = linalg.norm ( pattern, 2 )  # Eucledian or 2 norm
    norm_pattern = pattern / coeff

    # print(norm_pattern)
    return norm_pattern


def convergePoint(pattern, base_pattern):
    r, c = pattern.shape

    thresh_space = list ( np.linspace ( 0, 1, num=50 ) )
    y = thresh_space

    i = 0

    thresh = 0

    thresh_pattern = pattern
    # thresh = thresh_space[i]

    # print(thresh_pattern)

    while not np.allclose ( base_pattern, thresh_pattern ):
        thresh = thresh_space[i]
        # print(np.allclose(base_pattern , thresh_pattern))
        # print(thresh)
        # print(pattern)
        thresh_pattern = np.where ( pattern > thresh, 1, 0 )

        i += 1

        # thresh += 0.3
        # print(i)

    # print('Converged at ' , thresh)

    # plt.plot(thresh_space , y , color = 'indigo' , label = 'Domain')
    # plt.scatter(thresh , thresh , c = 'black' , s = 50 , label = 'Convergence Point')
    # plt.title('Convergence point of Learning')
    # plt.show()

    return (thresh_pattern, thresh)


def activate(thresh, pattern):
    pattern = np.where ( pattern > thresh, 1, 0 )

    return pattern


def learnHopfield(pattern):
    hop = train ( pattern )

    return hop


def getPatternLabels(pattern, data_labels):
    x, y = np.where ( pattern == 1 )

    x = x.tolist ()
    y = y.tolist ()

    x_l = [data_labels[i] for i in x]
    y_l = [data_labels[j] for j in y]

    pattern_labels = sorted ( set ( zip ( x_l, y_l ) ) )

    return pattern_labels


def recall(base_pattern, test_pattern, hop, t, data_labels):
    r, c = test_pattern.shape
    mem_pat_acc = 0
    # print(r , c)

    result = predict ( test_pattern, hop )

    result = result.reshape ( (r, c) )

    r_x, r_y = np.where ( result == 1 )

    result_patterns = getPatternLabels ( np.triu ( result, k=0 ), data_labels )

    for b in base_pattern:
        base = b.reshape ( (r, c) )
        orig_patterns = getPatternLabels ( np.triu ( base, k=0 ), data_labels )
        test_pats = getPatternLabels ( np.triu ( test_pattern, k=0 ), data_labels )

        # print('Pattern in memory... ')
        # print(base)
        # print()
        # print('Test Pattern...' , t)
        # print(test_pattern)
        # print()
        # print('Result...')
        # print(result)
        print ( 'Test Patterns: ', test_pats )
        print ( 'Result Patterns: ', result_patterns )
        print ( 'Memory Orig Patterns: ', orig_patterns )

        pat_not_trained = set ( result_patterns ).difference ( orig_patterns )
        test_pat_not_trained = set ( test_pats ).difference ( result_patterns )

        test_pat_acc = (abs ( len ( test_pats ) - len ( test_pat_not_trained ) ) / len ( test_pats )) * 100

        if len ( result_patterns ) == 0:
            mem_pat_acc = 0

        else:
            mem_pat_acc = (abs ( len ( orig_patterns ) - len ( pat_not_trained ) ) / len ( orig_patterns )) * 100

        print ( 'Patterns Lost: ', test_pat_not_trained if len ( test_pat_not_trained ) > 0 else 'None' )
        print ( 'New Patterns added : ', pat_not_trained if len ( pat_not_trained ) > 0 else 'None' )

        # res_sim = np.sum(base == result)

        # res_acc = ((abs((r * c) - res_sim) / (r * c)) * 100)

        # test_sim = np.sum(base == test_pattern)
        # print(test_sim)

        # test_acc = ((abs((r * c) - test_sim) / (r * c)) * 100)

        print ( 'Recalling Pattern..... ', t )
        print ( 'Test - Memory cos similarity is : ', calcCosineSimilarity ( base, test_pattern ) )
        print ( 'Result - Memory cos similarity is : ', calcCosineSimilarity ( base, result ) )
        print ( 'Test - Result pattern recovery acc : ', test_pat_acc )
        print ( 'Memory - Result pattern recovery acc : ', mem_pat_acc )
        print ()

    # stored_pattern = hop.get_stored_patterns()

    # print(stored_pattern , stored_pattern.shape)


data_labels = generateDataSets ()
n_cols = len ( data_labels )

base_pattern = np.zeros ( (n_cols, n_cols), dtype=np.int )
usage_pattern = np.random.randint ( 5, 60, (n_cols, n_cols) )
usage_pattern = usage_pattern.astype ( np.dtype ( float ) )

train_links = ['AB', 'BE', 'BC', 'AD']

base_pattern = createUsagePattern ( base_pattern, data_labels, train_links, train_str='base' )

trained_pattern = createUsagePattern ( usage_pattern, data_labels, train_links )

# print(base_pattern)
norm_pattern = normalize_mat ( trained_pattern )
thresh_bin_pattern, thresh = convergePoint ( norm_pattern, base_pattern )

# print(thresh_bin_pattern)


test_rel = createTestPatterns ( data_labels, n_cols )
test_pattern = np.random.randint ( 5, 60, (n_cols, n_cols) )
test_pattern = test_pattern.astype ( np.dtype ( float ) )

# print(test_rel)
tot_patterns = []
tot_weight = []
t_abc = np.array ( [] )
temp = np.array ( [] )
wt = np.ones ( (n_cols * n_cols, n_cols * n_cols) )

# tot_patterns.append(thresh_bin_pattern.reshape((1 , n_cols * n_cols)))

##i = 0
##while i < 3:
##      for t in test_rel[i : i + 1]:
##            base_pattern = np.zeros((n_cols , n_cols) , dtype = np.int)
##            base_pattern = createUsagePattern(base_pattern , data_labels , [t] , train_str = 'base')
##
##            test_pattern = createUsagePattern(test_pattern , data_labels , [t])
##            norm_test_pattern = normalize_mat(test_pattern)
##            test_bin_pattern , thresh = convergePoint(norm_test_pattern , base_pattern)
##
##            cs = calcCosineSimilarity(thresh_bin_pattern , test_bin_pattern)
##
##            cs = float(cs)
##            #print('CS: ' , cs)
##            #plt.plot(test_bin_pattern.flatten() , norm_test_pattern.flatten())
##            #plt.show()
##
##            print('Learning Pattern: ' , t , ' converged at: ' , thresh , ' cos sim: ' , cs)
##
##            if cs < 0.69:
##
##                  r ,c = test_bin_pattern.shape
##                  temp = test_bin_pattern.reshape((1 , r * c))
##                  tot_patterns.append(temp)
##
##            #if t == 'ABC':
##                  #t_abc = temp
##
##
##            test_pattern = np.random.randint(5 , 60 , (n_cols , n_cols))
##            test_pattern = test_pattern.astype(np.dtype(float))
##
##
##      #tot_test_patterns = np.concatenate([t for t in tot_patterns])
##
##      #print(temp , i)
##
##      hop = learnHopfield(temp)
##      #print('Weight Array')
##      #print(hop.get_stored_patterns())
##      wt = hop.get_stored_patterns()
##      tot_weight.append(wt)
##      i += 1
##
##
###print(tot_weight)
##
##tw = np.sum(tot_weight , axis = 0)
##hop.set_weight(tw)
##print()
##
##print(tw)
##
##print()
##print()


for t in test_rel[0: 1]:
    base_pattern = np.zeros ( (n_cols, n_cols), dtype=np.int )
    base_pattern = createUsagePattern ( base_pattern, data_labels, [t], train_str='base' )

    test_pattern = createUsagePattern ( test_pattern, data_labels, [t] )
    norm_test_pattern = normalize_mat ( test_pattern )
    test_bin_pattern, thresh = convergePoint ( norm_test_pattern, base_pattern )

    cs = calcCosineSimilarity ( thresh_bin_pattern, test_bin_pattern )

    cs = float ( cs )

    print ( test_bin_pattern )
    print ( thresh_bin_pattern )

    print ( 'Learning Pattern: ', t, ' converged at: ', thresh, ' cos sim: ', cs )

    if cs < 0.69:
        r, c = test_bin_pattern.shape
        temp = test_bin_pattern.reshape ( (1, r * c) )
        tot_patterns.append ( temp )

        # if t == 'ABC':
        # t_abc = temp

    test_pattern = np.random.randint ( 5, 60, (n_cols, n_cols) )
    test_pattern = test_pattern.astype ( np.dtype ( float ) )

    tot_test_patterns = np.concatenate ( [t for t in tot_patterns] )

hop = learnHopfield ( tot_test_patterns )
##print(hop.get_stored_patterns())
##wt = hop.get_stored_patterns()

# hop.set_weight(wt)

for t in test_rel[0: 1]:
    print ( 'Recalling pattern... : ', t )
    base_pattern = np.zeros ( (n_cols, n_cols), dtype=np.int )
    base_pattern = createUsagePattern ( base_pattern, data_labels, [t], train_str='base' )

    test_pattern = createUsagePattern ( test_pattern, data_labels, [t] )

    norm_test_pattern = normalize_mat ( test_pattern )
    # print(norm_test_pattern)

    test_bin_pattern, thresh = convergePoint ( norm_test_pattern, base_pattern )
    # print(test_bin_pattern)

    recall ( tot_patterns, test_bin_pattern, hop, t, data_labels )
    test_pattern = np.random.randint ( 5, 60, (n_cols, n_cols) )
    test_pattern = test_pattern.astype ( np.dtype ( float ) )

# print('Total trained patters: ' , len(test_rel))