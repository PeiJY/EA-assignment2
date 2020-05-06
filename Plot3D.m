  %{     close all;
        clear;
        date = load("log.txt");
        
        x=date(:,1);
        y=date(:,2);
        z=date(:,3);
        scatter3(x,y,z,'.')
        %}

        
        x=date(:,1);
        y=date(:,2);
        fes=date(:,3);
        evalue = date(:,4)
        scatter3(x,y,evalue,50,fes,'.')