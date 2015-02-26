# see http://www.gnu.org/prep/standards/html_node/Makefile-Conventions.html
SHELL = /bin/sh

#### System Configuration ####

BOOST = ~/programming/boost/boost_1_57_0
CC = g++
CDEBUG = -g3 -Wall
CFLAGS = $(CDEBUG) -O -std=c++0x
ALL_CFLAGS = -I$(BOOST) $(CFLAGS)

#defining paths
prefix := .
bindir := $(prefix)/bin
libdir := $(prefix)/lib
srcdir := $(prefix)/source
testdir := $(prefix)/test

#### End of system configuration section ####


# find all .cpp files using GNU wildcard
_SRC1 := $(wildcard $(srcdir)/*.cpp)
_SRC2 := $(wildcard $(srcdir)/utility/*.cpp)
_SRC3 := $(wildcard $(testdir)/*.cpp)
# collect source files
SRC := $(_SRC1) $(_SRC2) $(_SCR3)

# templates and (virtual) base classes (.h files)
HEAD := $(testdir)/test_template.h $(srcdir)/Classifier.h

# collect object files and prepand library path
_OBJ := BayesClassifier.o DensityTreeClassifier.o CopulaClassifier.o \
        DensityForestClassifier.o CopulaForestClassifier.o \
        common.o node_t.o test_template.o splits.o
OBJ  := $(patsubst %,$(libdir)/%,$(_OBJ))


# phony targets can always be called
.PHONY: all help clean
#.INTERMEDIATE: $(OBJ)

# target: all - Default target. Install everything.
all: install test

# target: install - Install the classifier package.
install: _installdirs build classifier


# Make sure all installation directories (e.g. $(bindir))
# actually exist by making them if necessary.
_installdirs: mkinstalldirs
	@./mkinstalldirs $(bindir) $(libdir)

# target: build - Compile object files.
build: $(HEAD) $(SRC) $(OBJ)

# automatic variables, e.g.:
# $@ LHS of : i.e. target
# $< $^ RHS of : i.e. 1 prerequisite or all

# pull in dependency info for *existing* .o files
#-include $(OBJ:.o=.d)

# compilation rule for objects from source using pattern matching i.e. %
# creating mkfiles whith dependency info via -MM
$(libdir)/%.o: $(srcdir)/%.cpp $(srcdir)/%.h
	$(CC) $(ALL_CFLAGS) -c -o $@ $<
	$(CC) $(ALL_CFLAGS) -MM $< > $(libdir)/$*.d

# compile objects from source/utitlity
$(libdir)/%.o: $(srcdir)/utility/%.cpp $(srcdir)/utility/%.h
	$(CC) $(ALL_CFLAGS) -c -o $@ $<
	$(CC) $(ALL_CFLAGS) -MM $< > $(libdir)/$*.d

# compile objects from test, i.e. test_template
$(libdir)/%.o: $(testdir)/%.cpp $(testdir)/%.h
	$(CC) $(ALL_CFLAGS) -c -o $@ $<
	$(CC) $(ALL_CFLAGS) -MM $< > $(libdir)/$*.d

# target: test - Test suit.
test: test_depth test_nearestneighbors test_readin test_sortindex \
      test_getcopula

# target: classifier - Classifier suit.
classifier: test_bayes test_densitytree test_densityforest \
            test_copula test_copulaforest

# general rule for creating executable binaries
test_%: $(testdir)/test_%.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: test_bayes - Naive BayesClassifier & Generator.
test_bayes: $(testdir)/test_bayes.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: test_densitytree - DensityTreeClassifier & Generator.
test_densitytree: $(testdir)/test_densitytree.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: test_copula - CopulaClassifier & Generator.
test_copula: $(testdir)/test_copula.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: test_densityforest - DensityForestClassifier & Generator.
test_densityforest: $(testdir)/test_densityforest.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: test_copulaforest - CopulaForestClassifier & Generator.
test_copulaforest: $(testdir)/test_copulaforest.cpp $(OBJ)
	$(CC) $(ALL_CFLAGS) -o $(bindir)/$@ $^

# target: help - Display callable targets. Debug Output.
help:
	@egrep "^# target:" [Mm]akefile
	@echo Objects: $(OBJ)

# target: clean - Delete created files.
clean:
	@rm -f $(libdir)/*.o $(libdir)/*.d $(bindir)/*