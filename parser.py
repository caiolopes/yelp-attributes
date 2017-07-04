import json 
import csv
import pandas as pd


def business_parser(min_review_count=50):
    # English cities
    businesses = []
    cities = ['Pittsburgh', 'Charlotte', 'Urbana-Champaign', 'Phoenix', 'Las Vegas', 'Madison', 'Cleveland']
    my_attrs = ['Alcohol', 'Delivery', 'GoodForKids', 'NoiseLevel', 'WheelchairAccessible', 'WiFi', 'BikeParking', 'RestaurantsTakeOut']

    with open('data/yelp_academic_dataset_business.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            categories = data['categories']
            if (data['city'] in cities and
                    data['review_count'] >= min_review_count and
                    categories is not None and
                    'Restaurants' in data['categories']):
                attrs = data['attributes']
                # there are a few some business without attributes
                if attrs is not None:
                    business = {'id': data['business_id'], 'stars': data['stars'], 'review_count': data['review_count']}
                    for attr in my_attrs:
                        attr_class = '?'
                        for a in attrs:
                            if attr in a:
                                attr_class = a.split(':')[1].strip()
                                break
                        business[attr] = attr_class

                    businesses.append(business)

        print('Total business:', len(businesses))
        print('Generating business.csv file...')
        f = open('data/business.csv', 'w')
        # header
        print('id,business_id,stars,review_count,', ','.join(map(str, my_attrs)), sep='', file=f)
        for i, v in enumerate(businesses):
            attrs = []
            attrs.append(i+1)
            attrs.append(v['id'])
            attrs.append(v['stars'])
            attrs.append(v['review_count'])
            for attr in my_attrs:
                if v[attr] == '?':
                    attrs.append('')
                else:
                    attrs.append(v[attr])

            print(','.join(map(str, attrs)), file=f)


def review_parser():
    df = pd.read_csv('data/business.csv').dropna()
    # converts business data to hash table of
    # {'attr': { 'business_id': 'value', ...}, ...}
    business = df.set_index('business_id').to_dict()
    business.pop('id', None)

    labels = ['business_id', 'text']
    labels.extend([key for key in business.keys()])

    writer = csv.writer(open('reviews.csv', 'w', newline="\n", encoding="utf-8"), quoting=csv.QUOTE_ALL, quotechar='"')
    writer.writerow(labels)
    with open('data/yelp_academic_dataset_review.json', 'r') as f:
        for line in f:
            review = json.loads(line)
            business_id = review['business_id']
            attrs = [business_id, review['text'].replace('\n', ' ').replace('\r', '').strip()]
            for label in labels:
                try:
                    attrs.append(business[label][business_id])
                except KeyError:
                    pass
            if len(attrs) > 2:
                writer.writerow(attrs)



business_parser()
review_parser()
