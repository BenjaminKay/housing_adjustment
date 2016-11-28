function [ sched ] = getSchedule(account,ppn,queue,time,DataLocation,ClusterMatlabRoot)
sched = findResource('scheduler','type','generic');
set(sched,'HasSharedFilesystem',true);
set(sched,'DataLocation',DataLocation);
set(sched,'ClusterOsType','unix');
set(sched,'getJobStateFcn',@getJobStateFcn);
set(sched,'destroyTaskFcn',@destroyJobFcn);
set(sched,'ClusterMatlabRoot',ClusterMatlabRoot);
set(sched,'SubmitFcn',{@distributedSubmitFcn,account,time,queue,ppn});
set(sched,'ParallelSubmitFcn',{@parallelSubmitFcn,account,time,queue,ppn});
