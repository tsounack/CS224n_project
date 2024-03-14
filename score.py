import pandas as pd
import ast

class score:
    """
    score is a class that is built to evaluate an information retrieval schema, whether that be bm25 algorithim, cosine similarity or bm25 
    with MuGI

    Methods:
        
    """

    def __init__(self, csv_path: str, df=None):
        """
        Initializes a new instance of the MuGI class.

        Args:
            csv_path (str): a path of a csv to be evaluated 
            df (dataframe): a data frame 
        """
        if df == None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = df
    
    def compute_score(self):
        """
        Computes the score of each of infomration retrieval column and creates a dictionary of the method to the score 

        Args:
            None

        Returns: 
            score_dictionary (Dict[str, int]): A dictionary of columns to scores 
        """
        score_dictionary = {}
        for index, row in self.df.iterrows():
            gold_page = row['Page']
            for column_name, value in row.items():
                if 'k=' in column_name:
                    value = ast.literal_eval(value)
                    if gold_page in value: 
                        index = value.index(gold_page)
                        length = len(value)
                        score_dictionary[column_name] = score_dictionary.get(column_name, 0) + (length - index) / length / len(self.df)
        return score_dictionary
    def compute_precision(self):
        """
        Computes the precision of each information retrieval column and creates a dictioanry of the method to the precision relative to each index

        Args:
            None

        Returns:
            score_dictionary (Dic[str, List[int]]): a diciotnary of columns to a list of precision of each index 
        """
        score_dictionary = {}
        for index, row in self.df.iterrows():
            gold_page = row['Page']
            for column_name, value in row.items():
                if 'k=' in column_name:
                    value = ast.literal_eval(value)
                    for i in range(len(value)):
                        if gold_page in value[:i+1]:
                            lst = [0] * len(value)
                            lst[i] = 1
                            score_dictionary[column_name] = [a + b / len(self.df) for a, b in zip(score_dictionary.get(column_name, [0, 0, 0]), lst)]
        return score_dictionary


