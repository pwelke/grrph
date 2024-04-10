import numpy as np

def kernel_alignment(g1, g2):
    ''' see, e.g. Equation (1) in 
    Tinghua Wang, Dongyan Zhao, Shengfeng Tian: 
    An overview of kernel alignment and its applications.
    Artif Intell Rev (2015) 43:179–192
    DOI 10.1007/s10462-012-9369-4
    https://link.springer.com/content/pdf/10.1007%2Fs10462-012-9369-4.pdf '''
    g1 = g1.flatten()
    g2 = g2.flatten()
    
    f11 = np.dot(g1, g1)
    f22 = np.dot(g2, g2)
    f12 = np.dot(g1, g2)
    
    return f12 / np.sqrt(f11 * f22)


def max_normalize_gram(gram):
    return gram / np.max(np.abs(gram))