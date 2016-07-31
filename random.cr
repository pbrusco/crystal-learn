module Random
  def Random.sequence(n : Int32)
    seq = [] of Float64
    n.times do
      seq << rand() 
    end
    seq
  end
end
