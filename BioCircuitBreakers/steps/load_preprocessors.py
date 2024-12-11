import preprocessors

def LoadPreprocessors(proj):
    preprocessor_list = []
    for ppc in proj.preprocessors:
        preprocessor = getattr(preprocessors, ppc)
        preprocessor_list.append(preprocessor(proj))

    return preprocessor_list