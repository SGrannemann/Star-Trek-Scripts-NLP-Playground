# this returns all tokens in the vocabulary
tfidf_features = tfidf_model.get_feature_names()


# this creates a string that shows all coefficients and the token for the first dimension.
test = " ".join([
    "%+0.3f*%s" % (coef, feat) 
    for coef, feat in zip(svd_model.components_[0], tfidf_features)
])


# get 10 best features for dimension 1
# argsort returns indices in the order that would sort the components_ array in ascending order. [::-1] loops over that array from the last item onwards
best_features = [tfidf_features[i] for i in svd_model.components_[0].argsort()[::-1]]