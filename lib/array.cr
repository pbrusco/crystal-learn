class Array(T)
  def shape
    get_shape(self)
  end

  def [](positions : Array(Int))
    res = [] of T
    positions.each do |idx|
      res << self[idx]
    end
    res
  end

  def take(n : Int)
    self[0..n - 1]
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
    max_id = self.map { |x| self.count(x) }.argmax
    self[max_id]
  end

  def frequencies
    diff_values = self.uniq
    count = self.size
    frequencies = diff_values.map { |v| {v, self.count(v).to_f / count} }
  end

  def indices_of(elem)
    indices = [] of Int32
    self.each_with_index { |x, i|
      indices << i if x == elem
    }
    indices
  end

  def select_with_indices(&block)
    selected = [] of T
    indices = [] of Int32
    self.each_with_index { |x, i|
      if yield(x)
        selected << x
        indices << i
      end
    }
    {selected, indices}
  end

  def mean
    self.sum / self.size.to_f
  end

  def std
    mu = self.mean
    Math.sqrt(self.map { |x| (x - mu).abs2 }.sum / self.size.to_f)
  end

  # TODO: se puede pedir por tipos que self sea tipo Array(Array(T))?
  def mean(*, by)
    case by
    when :column
      self.transpose.mean(by: :row)
    when :row
      self.map(&.mean)
    else
      raise "unknown parameter for mean (by = #{by})"
    end
  end

  def std(*, by)
    case by
    when :column
      self.transpose.std(by: :row)
    when :row
      self.map(&.std)
    else
      raise "unknown parameter for std (by = #{by})"
    end
  end

  def substract(other, *, by)
    case by
    when :column
      substract_by_column(self, other)
    when :row
      substract_by_row(self, other)
    else
      raise "invalid parameter"
    end
  end

  def divide(numbers : Array(Number), *, by)
    case by
    when :column
      self.transpose.zip(numbers).map { |x, y| x / y }.transpose
    when :row
      self.zip(numbers).map { |x, y| x / y }
    else
      raise "invalid parameter"
    end
  end

  def /(number : Number)
    self.map { |x| x / number }
  end

  def keep_columns(colum_numbers)
    self.transpose[colum_numbers].transpose
  end
end

def get_shape(arr : Array(Array(T))) forall T
  {arr.size, arr[0].size}
end

def get_shape(arr : Array(T)) forall T
  arr.size
end

def substract_by_column(matrix : Array(Array(Number)), y : Array(Number))
  matrix.transpose.zip(y).map { |column, y| substract(column, y) }.transpose
end

def substract_by_row(matrix : Array(Array(Number)), y : Array(Number))
  matrix.zip(y).map { |row, y| substract(row, y) }
end

def substract(arr : Array(Number), y : Number)
  arr.map { |x| x - y }
end
