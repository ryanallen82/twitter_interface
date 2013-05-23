import twitter
import json
import re
import math
import sqlite3
from collections import Counter


# Global variables
from twitter_config import CONSUMER_KEY,CONSUMER_SECRET,OAUTH_TOKEN,OAUTH_TOKEN_SECRET

import feedfilter


# Returns a Twitter API token
def getApi():
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,CONSUMER_KEY, CONSUMER_SECRET)
    twitter_api = twitter.Twitter(domain='api.twitter.com', api_version='1.1',auth=auth)
    return twitter_api

# Takes Twitter APU token, prints world trends
def getWorldTrends(twitter_api):
    WORLD_WOE_ID = 1
    world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
    print json.dumps(world_trends, indent=1)

# Takes Twitter API token, a string keyword/phrase, an integer count, and the
# number of loops as an integer. Returns tweets containing keyword/phrase.
def searchStatuses(twitter_api,keyword,count,loops):

    search_results = twitter_api.search.tweets(q=keyword, count=count)
    statuses = search_results['statuses']

    for _ in range(loops):
        try:
            next_results=search_results['search_metadata']['next_results']
            kwargs = dict([kv.split('=') for kv in next_results[1:].split("&")])
            search_results=twitter_api.search.tweets(**kwargs)
            statuses += search_results['statuses']
        except:
            pass

    tweets = [ status['text'] for status in statuses]

    return tweets

# Takes a list of tweets and prints total words, unique words, diversity,
# and average words per tweet
def simpleAnalysis(tweets):
    words = []
    for t in tweets:
        words += [w for w in t.split()]

    print ""
    print "-------Simple Analysis-------"
    print "Total words: %s" % (len(words))

    print "Unique words: %s" % (len(set(words)))

    print "Lexical diversity: %s" % (1.0*len(set(words))/len(words))

    print "Average words per tweet: %s " % (1.0*sum([len(t.split()) for t in tweets])/len(tweets))

# Takes a list of tweets and prints the 50 most common words
def freqAnalysis(tweets):
    words = []
    for t in tweets:
        words+=[w for w in t.split()]

    counter=Counter(words)
    print ""
    print "-------50 Most Common Words-------"
    for item in counter.most_common(50):
        print item[0], ', '.join(map(str, item[1:]))

# Takes a list of tweets and prints statuses
def printStatuses(tweets):
    for i in range(len(tweets)):
        print tweets[i]

# Takes a list of tweets and print retweets
def reTweets(tweets):
    rt_patterns = re.compile(r"(RT|via)((?:\b\W*@\w+)+)", re.IGNORECASE)
    for t in tweets:
        for item in rt_patterns.findall(t):
            print item[0], item[1], t


# Takes a list of tweets and returns the unique set of words only
def getwords(tweets):
    words=[]
    for t in tweets:
        words+=[w.encode('utf-8') for w in t.split()]
    return words

def sampletrain(cl):
    cl.train('Nobody owns the water.','good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now','bad')
    cl.train('make quick money at the online casino','bad')
    cl.train('the quick brown fox jumps','good')

class classifier:

    def __init__(self,getfeatures,filename=None):
        # Counts of feature/category combinations
        self.fc={}
        # Counts of documents in each category
        self.cc={}
        self.getfeatures=getfeatures

    def setdb(self,dbfile):
        self.con=sqlite3.connect(dbfile)
        self.con.text_factory = str
        self.con.execute('CREATE TABLE IF NOT EXISTS fc(feature,category,count)')
        self.con.execute('CREATE TABLE IF NOT EXISTS cc(category,count)')

    # Increase the count of a feature/category pair
    def incf(self,f,cat):
        count=self.fcount(f,cat)
        if count==0:
            sql=("INSERT INTO fc (feature, category,count) VALUES(?,?,?)")
            self.con.execute(sql, (f,cat,1))
        else:
            self.con.execute("UPDATE fc SET count=%d WHERE feature='%s' AND category='%s'" % (count+1,f,cat))


    # Increase the count of a category
    def incc(self,cat):
        count=self.catcount(cat)
        if count==0:
            self.con.execute("INSERT INTO cc VALUES ('%s',1)" % (cat))
        else:
            self.con.execute("UPDATE cc SET count=%d WHERE category='%s'" % (count+1,cat))

    # The number of times a feature has appeared in a category
    def fcount(self,f,cat):
        sql=("SELECT count FROM fc WHERE feature = ? AND category = ?")
        res=self.con.execute(sql, (f,cat)).fetchone();
        if res==None: return 0
        else: return float(res[0])

    # The number of items in a category
    def catcount(self,cat):
        res=self.con.execute('SELECT count FROM cc WHERE category="%s"' % (cat)).fetchone()
        if res==None: return 0
        else: return float(res[0])

    # The total number of items
    def totalcount(self):
        res=self.con.execute('SELECT SUM(count) from cc').fetchone();
        if res==None: return 0
        return res[0]

    # The list of all categories
    def categories(self):
        cur=self.con.execute('SELECT category FROM cc');
        return [d[0] for d in cur]

    # Takes an item and a classification and breaks the item into different
    # features. Call incf to increase classification count for each feature,
    # then increases the total count for given classification
    def train(self,item,cat):
        features=item.split()
        # Increment the count for every feature with this category
        for f in features:
            self.incf(f,cat)

        # Increment the count for this category
        self.incc(cat)

        self.con.commit()

    # Returns conditional probability
    def fprob(self,f,cat):
        if self.catcount(cat)==0: return 0

        # The total number of times this feature appeared in this catagory
        # divided by the total number of items in this category
        return self.fcount(f,cat)/self.catcount(cat)

    def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
        # Calculate current probability
        basicprob=prf(f,cat)

        # Count the number of times this feature has appeared in all categories
        totals=sum([self.fcount(f,c) for c in self.categories()])

        # Calculate the weighted average
        bp=((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp


# Fisher classifier
class fisher(classifier):
    def cprob(self,f,cat):
        # The frequency of this feature in this category
        clf=self.fprob(f,cat)
        if clf==0: return 0

        # The frequency of this feature in all the categories
        freqsum=sum([self.fprob(f,c) for c in self.categories()])

        # The probability is the frequency in this category divided by the
        # overall frequency
        p=clf/(freqsum)

        return p

    def fisherprob(self,item,cat):
        # Multiply all the probabilities together
        p=1
        features=self.getfeatures(item)
        for f in features:
            p*=(self.weightedprob(f,cat,self.cprob))

        # Take the natural log and multiply by -2
        fscore=-2*math.log(p)

        # Use the inverse chi2 function to get a probability
        return self.invchi2(fscore,len(features)*2)

    def invchi2(self,chi,df):
        m = chi/2.0
        suma=term=math.exp(-m)
        for i in range(1, df//2):
            term*=m/i
            suma+=term
        return min(suma,1.0)

    def __init__(self,getfeatures):
            classifier.__init__(self,getfeatures)
            self.minimums={}

    def setminimum(self,cat,mini):
        self.minimums[cat]=mini

    def getminimum(self,cat):
        if cat not in self.minimums: return 0
        return self.minimums[cat]

    def classify(self,item,default=None):
        # Loop through looking for the best result
        best = default
        maxa=0.0
        for c in self.categories():
            p=self.fisherprob(item,c)
            # Make sure it exceeds its minimum
            if p>self.getminimum(c) and p>maxa:
                best=c
                maxa=p
        return best


