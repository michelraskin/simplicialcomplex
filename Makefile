CXX = g++

CXXFLAGS = -Wall -Wextra -std=c++20

TARGET = simplicial_complex

SRC = SimplexMain.cpp
DEPS = Simplex.hpp SimplicialComplex.hpp
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