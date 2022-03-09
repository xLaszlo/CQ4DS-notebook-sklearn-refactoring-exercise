
switch=$1

if [ -z "$switch" ]; then
    echo "use ./create_branch -do"
else
    cp -r Slide2 Slide1
    cp -r Slide3 Slide2
    cp -r Slide4 Slide3
    cp -r Slide5 Slide4
fi




