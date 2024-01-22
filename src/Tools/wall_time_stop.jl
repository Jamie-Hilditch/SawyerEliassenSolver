struct WallTimeStop
    wall_time_stop :: UInt64 
    start :: UInt64
    stop :: UInt64
    

    function WallTimeStop(wall_time_stop)
        start = time_ns()
        new(wall_time_stop,start,start + wall_time_stop)
    end
end

function (wts :: WallTimeStop)()
    time_ns() > wts.stop
end