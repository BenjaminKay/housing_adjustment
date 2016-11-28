function run_opt_on_sdsc_110829b()
processors=32
ppn=8
account='bkay'
queue='batch'
%hours, minutes, seconds
time='15:00:00' 
DataLocation='/home/bkay/matlab/'
matlabRoot='/home/beta/matlab.2011a'
sched = getSchedule(account,ppn,queue,time,DataLocation,matlabRoot)
j = createMatlabPoolJob(sched);
j.FileDependencies={'run_opt_on_sdsc_110829b.m','getSchedule.m','HouseStateProcessor_110824.m', ...
                    'flavinopt_110829b.m','CheckPolicy_110822.m','OptimizeValueSct_110822.m', ...
                    'beninterp2_for_mat.mexa64','meshvecs.m','howard_until_converge_110824.m', ...
                    'beninterp1_for_mat.mexa64','OptimizeValueSct_continuous_110822.m'};
set(j,'MinimumNumberofWorkers',processors);
set(j,'MaximumNumberofWorkers',processors);
                %  Optimization fcn,# outputs, input args
t = createTask(j,@flavinopt_110829b,6,{1500,113,1,500,1000})
set(t,'CaptureCommandWindowOutput',true);
submit(j)
j.waitForState
o=j.getAllOutputArguments
o{:}
t.CommandWindowOutput
