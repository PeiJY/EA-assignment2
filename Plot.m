        close all;
        clear;
        date = load("logs\problem003run001_generation_log.txt");
        figure(1)
        indiv=date(:,3);
        fes=date(:,2);
        fitness=date(:,1);

        scatter3(indiv,fes,fitness,1,fes,'.')
        xlabel('indiv')
        ylabel('fes')
        zlabel('fitness')


        figure(3)
        data =  load("average\problem001_opts_log.txt");
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