require "spec"
require "./ml"
require "./trees"

describe ML do
  describe "entropy" do
    it "should calculate tags entropy" do
      ML.entropy(tags: [1,1,1,1,1,1,1,1,1,0,0,0,0,0]).should be_close(0.94, delta: 0.01)
    end

    it "should calculate tags entropy given categorical feature" do
      ML.entropy(tags: [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                 given: [:s, :s, :s, :o, :o, :o, :o, :r, :r, :s, :s, :r, :r, :r]
                 ).should be_close(0.693, delta: 0.01)
    end

  end
end
