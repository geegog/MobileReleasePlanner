import pandas as pd
import numpy as np

from google_play_scraper import Sort, reviews


class PlayStore:

    def fetch_reviews(self, app_id, count):
        result, continuation_token = reviews(
            app_id=app_id,
            count=count,
            sort=Sort.MOST_RELEVANT
        )
        return result, continuation_token


def main():
    ps = PlayStore()
    result, continuation_token = ps.fetch_reviews("org.catrobat.paintroid", 501)

    rows = []
    columns = ['reviewId', 'userName', 'userImage', 'content', 'score',
               'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt']

    for d in result:
        rows.append([d['reviewId'], d['userName'], d['userImage'], d['content'], d['score'],
                     d['thumbsUpCount'], d['reviewCreatedVersion'], d['at'], d['replyContent'], d['repliedAt']])

    df = pd.DataFrame(np.array(rows),
                      columns=columns)

    df.to_csv("org.catrobat.paintroid.csv")


if __name__ == "__main__":
    main()
