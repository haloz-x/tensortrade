from tensortrade.feed import Stream, NameSpace
from tensortrade.feed.core.feed import PushFeed

with NameSpace('test'):
    s1 = Stream.placeholder(dtype="float").rename("s1")
    s2 = Stream.placeholder(dtype="float").rename("s2")

    feed = PushFeed([
        s1.rename("v1"),
        s2.rename("v2")
    ])
    arr1 = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    arr2 = [-5, 5, -4, 4, -3, 3, -2, 2, -1, 1]
    for i in range(len(arr1)):
        output = feed.push({
            "test:/v1": arr1[i],
            "test:/v2": arr2[i]
        })
        print(output)
