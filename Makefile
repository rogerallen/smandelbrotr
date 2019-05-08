CXX=g++
CXXFLAGS=
LDFLAGS=
LDLIBS=-lsfml-graphics -lsfml-window -lsfml-system -lGL

SRCS=main.cpp window.cpp

OBJS=$(subst .cpp,.o,$(SRCS))

all: smandelbrotr

smandelbrotr: $(OBJS)
	$(CXX) $(LDFLAGS) -o smandelbrotr $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS) smandelbrotr .depend

include .depend
