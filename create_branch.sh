
switch=$1

if [ -z "$switch" ]; then
    echo "use ./create_branch -do"
else
    cp Slide2/* Slide1
    cp Slide3/* Slide2
    cp Slide4/* Slide3
    cp Slide5/* Slide4
fi




