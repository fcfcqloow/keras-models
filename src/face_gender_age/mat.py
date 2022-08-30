# https://github.com/imdeepmind/processed-imdb-wiki-dataset
import numpy as np
import datetime as date

from dataclasses import dataclass
from scipy.io import loadmat
from dateutil.relativedelta import relativedelta

@dataclass
class FaceFile:
    age:         int
    gender:      str
    path:        str
    face_score1: any
    face_score2: any

wiki_mat = 'dataset/file/wiki_crop/wiki.mat'
imdb_mat = 'imdb_crop/imdb.mat'
def getFaceFilePaths() -> list:
    def wiki() -> list:
        wiki_data        = loadmat(wiki_mat)
        wiki             = wiki_data['wiki']
        wiki_photo_taken = wiki[0][0][1][0]
        wiki_full_path   = wiki[0][0][2][0]
        wiki_gender      = wiki[0][0][3][0]
        wiki_face_score1 = wiki[0][0][6][0]
        wiki_face_score2 = wiki[0][0][7][0]
        
        wiki_path    = ['dataset/file/wiki_crop/%s' % (path[0]) for path in wiki_full_path]
        wiki_genders = ['male' if wiki_gender[n] == 1 else 'female' for n in range(len(wiki_gender))]
        wiki_dob     = [file.split('_')[2] for file in wiki_path]
        def parseAge(i: int):
            try:
                d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
                d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
                rdelta = relativedelta(d2, d1)
                diff = rdelta.years
            except Exception as ex:
                print(ex)
                diff = -1
            return diff

        wiki_age = [parseAge(i) for i in range(len(wiki_dob))]

        return list(
            filter(
                lambda f: f.face_score2 == 'nan' and  f.face_score1 != '-inf',
                [FaceFile(w[0], w[1], w[2], w[3], w[4]) 
                    for w in np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T]
            )
        )

    return wiki()
