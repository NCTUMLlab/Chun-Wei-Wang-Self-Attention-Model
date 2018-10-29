wget https://github.com/jiacheng-xu/vmf_vae_nlp/raw/master/data/yelp.zip
unzip yelp.zip
rm yelp.zip

mkdir data
mv yelp/train.txt data/
mv yelp/valid.txt data/
mv yelp/test.txt data/
rm -r yelp
rm -r __MACOSX
