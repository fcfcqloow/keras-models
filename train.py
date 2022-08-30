import sys

from src.face_gender_age.main import train as gender_age_train

def main(args):
    if len(args) == 1:
        return
    if args[1] == 'gender_age':
        gender_age_train()
        

if __name__ == '__main__':
    main(sys.argv)

    