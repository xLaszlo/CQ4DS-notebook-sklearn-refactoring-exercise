
switch=$1

if [ -z "$switch" ]; then
    echo "use ./create_branch -do"
else
    cp Slide02/* Slide01
    cp Slide03/* Slide02
    cp Slide04/* Slide03
    cp Slide05/* Slide04
    cp Slide06/* Slide05
    cp Slide07/* Slide06
    cp Slide08/* Slide07
    cp Slide09/* Slide08
    cp Slide10/* Slide09
    cp Slide11/* Slide10
    cp Slide12/* Slide11
    cp Slide13/* Slide12
fi
