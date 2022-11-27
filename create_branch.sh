
switch=$1

if [ -z "$switch" ]; then
    echo "use ./create_branch -do"
else
    cp Step02/* Step01
    cp Step03/* Step02
    cp Step04/* Step03
    cp Step05/* Step04
    cp Step06/* Step05
    cp Step07/* Step06
    cp Step08/* Step07
    cp Step09/* Step08
    cp Step10/* Step09
    cp Step11/* Step10
    cp Step12/* Step11
    cp Step13/* Step12
    cp Step14/* Step13
    cp Step15/* Step14
    cp Step16/* Step15
    cp Step17/* Step16
    cp Step18/* Step17
    cp Step19/* Step18
    cp Step20/* Step19
    cp Step21/* Step20
    cp Step22/* Step21
fi
