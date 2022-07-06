wget -P ../weights_and_data/ https://www.robots.ox.ac.uk/~abrown/VFA_data/weights_and_data/person_images.zip
wget -P ../weights_and_data/ https://www.robots.ox.ac.uk/~abrown/VFA_data/weights_and_data/videos.zip 

unzip ../weights_and_data/person_images.zip -d ../weights_and_data/
unzip ../weights_and_data/videos.zip -d ../weights_and_data/

rm ../weights_and_data/person_images.zip
rm ../weights_and_data/videos.zip