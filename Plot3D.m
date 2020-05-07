        close all;
        clear;
        figure(1)
        date = load("logs\problem002run001_generation_log.txt");
        x=date(:,3);
        y=date(:,4);
        fes=date(:,1);
        evalue = date(:,2)
        scatter3(x,y,fes,50,evalue,'.')
        figure(2)
        scatter3(x,y,evalue,50,fes,'.')
        
        figure(3)
        data =  load("opts_log.txt");
        x = data(:,1);
        y = data(:,2);
        hold on;
        plot(x,y,'r');
        y = data(:,3);
        plot(x,y,'b');
        y = data(:,4);
        plot(x,y,'k');
        y = data(:,5);
        plot(x,y,'m');