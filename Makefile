# inspired by https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile/2481326
# that led to http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
CXX=g++
CXXFLAGS=
LDFLAGS=
LDLIBS=-lsfml-graphics -lsfml-window -lsfml-system -lGL

SRCS=main.cpp window.cpp

OBJS=$(subst .cpp,.o,$(SRCS))

DEPDIR := .depend
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

all: smandelbrotr

smandelbrotr: $(OBJS)
	$(CXX) $(LDFLAGS) -o smandelbrotr $(OBJS) $(LDLIBS)

clean:
	$(RM) $(OBJS) smandelbrotr
	$(RM) -r $(DEPDIR)

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS))))
