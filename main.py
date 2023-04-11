import warnings
warnings.simplefilter("ignore")

from model import get_answer
from pdftotext import pdftotext

def main():

    context = pdftotext()

    # question = "Motivated by the objective, what is our aim?"
    question = input("Please enter the question : \n")
    out = get_answer(question, context)
    print("The answer given by the model is : ")
    print(out)

    return None

if __name__=="__main__":
    main()




