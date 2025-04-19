CXX = g++

CXXFLAGS = -I /opt/homebrew/include/eigen3 -I /opt/homebrew/include/ -I ./gudhi-devel/src/Persistent_cohomology/include -I ./gudhi-devel/src/Simplex_tree/include -I ./gudhi-devel/src/common/include/ -I gudhi-devel/src/Rips_complex/include/ -Wall -Wextra -std=c++20 -O3 -flto -march=native -funroll-loops -ffast-math 

TARGET = simplicial_complex

SRC = SimplexMain.cpp GudhiTest.cpp
DEPS = Simplex.hpp SimplicialComplex.hpp NerveComplex.hpp SimplexUtils.hpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f $(OBJ) $(TARGET)

run: all
	./$(TARGET)