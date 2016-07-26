
class Array(T)
  def [](positions : Array(Int))
    res = [] of T
    positions.each do |idx|
      res << self[idx]
    end
    res
  end

  def take(n : Int)
    self[0..n-1]
  end

  def take(percentage : Float)
    n = (self.size * percentage).floor.to_i
    take(n)
  end

  def drop(n : Int)
    self[n..-1]
  end

  def drop(percentage : Float)
    n = (self.size * percentage).floor.to_i
    drop(n)
  end

  def argmax
    m = self.index(max).not_nil!
  end

  def mode
    max_id = self.map {|x| self.count(x)}.argmax
    self[max_id]
  end
end
