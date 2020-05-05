        close all;
        clear;
        date = load("log.txt");
        
        x=date(:,1);
        y=date(:,2);
        z=date(:,3);
        scatter3(x,y,z,'.')


