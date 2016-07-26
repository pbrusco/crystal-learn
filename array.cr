
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

  def frequencies
    diff_values = self.uniq
    count = self.size
    frequencies = diff_values.map {|v| {v, self.count(v).to_f / count}}
  end

  def indices_of(elem)
    indices = [] of Int32
    self.each_with_index {|x, i|
      indices << i if x == elem
    }
    indices
  end
end
