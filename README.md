# NFL-Tackle-Project
A MATLAB code that determines the Tackles Above Average for each player by position, a positions tackles, and the effectiveness on run vs pass plays.

The code draws from the NFL Data Bowl - public repository for player positional data, direction, speed, play indicators, and player indicators (Ballcarrier)

It uses the positional data to determine who the closest defender is to a ball carrier that isn't blocked and simulates a number of plays that a tackle occurs

By finiding instances when players make tackles and instnaces they fail, the program assigns a percent chance of a tackle being made at any instance. IT then determines the players expected numeber of tackles per season - and compares it to their total tackle number. IT then indicates the Tackles over expected - sorts the expected tackles and the over expected tackles byt position

It compares the tackles to the position average and determines each players tackles above the positional average. 
