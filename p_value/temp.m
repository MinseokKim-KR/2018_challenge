A1 = csvread('/home/mskim/autron/driver_data/170518/TW/170518_test1_TW/170518_test1_TW_L1.csv', 1);
A2 = csvread('/home/mskim/autron/driver_data/170518/TW/170518_test1_TW/170518_test1_TW_L2.csv', 1);
B1 = csvread('/home/mskim/autron/driver_data/170518/JH/170518_test1_JH/170518_test1_JH_L1.csv', 1);
B2 = csvread('/home/mskim/autron/driver_data/170518/JH/170518_test1_JH/170518_test1_JH_L2.csv', 1);
C1 = csvread('/home/mskim/autron/driver_data/170518/HH/170518_test1_HH/170518_test1_HH_L1.csv', 1);
C2 = csvread('/home/mskim/autron/driver_data/170518/HH/170518_test1_HH/170518_test1_HH_L2.csv', 1);
D1 = csvread('/home/mskim/autron/driver_data/170518/CS/170518_test1_CS/170518_test1_CS_L1.csv', 1);
D2 = csvread('/home/mskim/autron/driver_data/170518/CS/170518_test1_CS/170518_test1_CS_L2.csv', 1);
TW = [A1;A2];
JH = [B1;B2];
HH = [C1;C2];
CS = [D1;D2];
result = [];
j=0;
for i=1:442
    [h1, p1, ci1, stats1] = ttest2(TW(6000+2487611*i:2487611*(i+1)), JH(6000+2492908*i:2492908*(i+1)), 'Alpha', 0.1);
    [h2, p2, ci2, stats2] = ttest2(TW(6000+2487611*i:2487611*(i+1)), HH(6000+2487081*i:2487081*(i+1)), 'Alpha', 0.1);
    [h3, p3, ci3, stats3] = ttest2(TW(6000+2487611*i:2487611*(i+1)), CS(6000+2488458*i:2488458*(i+1)), 'Alpha', 0.1);
    [h4, p4, ci4, stats4] = ttest2(JH(6000+2492908*i:2492908*(i+1)), HH(6000+2487081*i:2487081*(i+1)), 'Alpha', 0.1);
    [h5, p5, ci5, stats5] = ttest2(JH(6000+2492908*i:2492908*(i+1)), CS(6000+2488458*i:2488458*(i+1)), 'Alpha', 0.1);
    [h6, p6, ci6, stats6] = ttest2(HH(6000+2487081*i:2487081*(i+1)), CS(6000+2488458*i:2488458*(i+1)), 'Alpha', 0.1);
    if (h1 ==1) && (h2 ==1) && (h3 ==1) && (h4 ==1) && (h5 ==1) && (h6 ==1)
       result = [result; i];
    end
end

disp("result")
disp(result)
disp(length(result))