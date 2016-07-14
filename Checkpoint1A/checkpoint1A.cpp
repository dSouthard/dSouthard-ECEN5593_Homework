
/* @file
 *  This is the developed Checkpoint-1A assignment for ECEN 5593, Summer 2016
 *  Author: Diana Southard
 *
 *
 * Overview: Use the PIN tool to instrument applications to implement a small
 * simulator to evaluate the prediction performance (accuracy) of a branch-
 * prediction buffer with various branch prediction schemes.
 *
 *  - Single-bit predictor: a 1-bit predictor storing (taken or not-taken last  
 * time)
 *  - Bi-model bit predictor: a 2-bit predictor saturating counter
 *  - Two-level (GAg) predictor using a 8-bit history register and a 2^8 entry 
 * pattern history table. Each pattern table entry has a 2-bit saturating counter.
 *  - Two-level (PAg) predictor using a 12-bit local branch history and a 2^12 
 * entry pattern history table. Each pattern table entry has a 2-bit saturating 
 * counter.
 *
 * For the 1-bit and 2-bit schemes, entries in the BTB tables are only initially 
 * created when a branch is first taken (non-taking branches will by default be  
 * predicted as non-taken since they have no previously created table entry). 
 * Global history for the GAg scheme is always updated whether the branch is in 
 * the BTB or if it is an always-taken branch. 
 *
 */

#include "pin.H"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <unistd.h> // for pid

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
static INT32 Usage()
{
	cerr << "This pin tool collects a profile of jump/branch/call instructions for an application\n";

	cerr << KNOB_BASE::StringKnobSummary();

	cerr << endl;
	return -1;
}
/* ===================================================================== */

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

// Variable: written once, written by pin tool, specify the output name, with usage comment in case of error
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE,         "pintool",
		"o", "output.out", "specify profile file name");

// Process id information, for multi-threaded/process program
// --> You can find a certain running processes: Written once, coming from the pin tool,
KNOB<BOOL>   KnobPid(KNOB_MODE_WRITEONCE,                "pintool",
		"i", "0", "append pid to output");

// limit how many branches simulated within the system: written once, coming from the pin tool,
KNOB<UINT64> KnobBranchLimit(KNOB_MODE_WRITEONCE,        "pintool",
		"l", "0", "set limit of dynamic branches simulated/analyzed before quit");

// Add knob to change which type of branch-prediction scheme is being used
// Variable: written once, coming from pin tool, specify the output name, with usage comment in case of error
KNOB<UINT64> KnobBranchPrediction(KNOB_MODE_WRITEONCE,         "pintool",
		"b", "0", "specify branch prediction scheme being used [0 = 1-bit, 1 = 2-bit, 2 = GAg, 3 = PAg, 4 = Hybrid (Combo of 2-bit and GAg) [defaults to 1-bit]");

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */
UINT64 CountSeen = 0;
UINT64 CountTaken = 0;
UINT64 CountCorrect = 0;
UINT64 CountMissed = 0;
UINT64 CountReplaced = 0;

// Enum class to keep track of which branch prediction scheme will be used,
// Default = 1-bit prediction, can be set via command line
enum PredictionType {
	ONE_BIT, TWO_BIT, GAg, PAg, HYBRID
};

// Variable to keep track of which prediction scheme is being used
PredictionType predictionType = ONE_BIT;

/* ===================================================================== */
/* Branch predictors                                                     */
/* ===================================================================== */
UINT64 mask = 0x03FF;
UINT64 GAgMASK = 0xFF;
UINT64 PAgMASK = 0xFFF;

#define BTB_SIZE 1024
#define BTB_GAG_TABLE_SIZE 256
#define BTB_PAG_TABLE_SIZE 4096

/* BTB = Branch Target Buffer
 * entry_bit:
 * 		Entry structures for the BTB Table
 * 		1-bit: uses valid, prediction, tag, and replaceCount variables
 * 		2-bit: also uses counter variable in addition to 1-bit variables
 * 		GAg: uses BTB_History, BTB_HistoryLength, and BTB_Table
 *
 */
struct entry_bit
{
	bool valid;						// Maintains track of valid entries in table
	bool prediction;				// Marks prediction choice for current entry
	UINT64 tag;						// Tag of current entry in table
	UINT64 ReplaceCount;			// Maintains count of total entries replaced
	UINT8 counter;					// Used for 2-bit prediction, keep track of current prediction counter
	UINT64 branchHistoryRegister;	// Used for PAg prediction, branch's individual history
}BTB[BTB_SIZE];


// History Variables for two-level branch predictions
UINT8 GlobalHistoryRegister;			// Global Branch History Register
//UINT8 HistorySize;					// Used to dynamically change the size of the HistoryPatternTables
UINT8 * HistoryPatternTable = NULL;		// Only initialized in the case of GAg/PAg prediction
UINT8 * PerAddr_HistoryTable = NULL;	// Only initialized in the case of PAg prediction
UINT8 hybridCounter;					// Used for Hybrid prediction

/* initialize the BTB */
VOID BTB_init()
{
	int i;
	for(i = 0; i < BTB_SIZE; i++)
	{
		// All entries are initially false
		BTB[i].valid = false;

		// All entry predictions are initially 'Not Taken' (false)
		BTB[i].prediction = false;

		// All entry tags are initially zeroed-out
		BTB[i].tag = 0;

		// Initialize ReplaceCount to 0
		BTB[i].ReplaceCount = 0;

		// Initialize counter (for 2-bit predictor) to 'Not Taken' (1)
		BTB[i].counter = 1;

		// Initialize branch history (for PAg)
		/* HINT: in the case of PAg, you will need to add a field to the BTB structure to
		 * maintain the history for the branch of that entry.
		 */
		BTB[i].branchHistoryRegister = 0x0;
	}

	// Initialize counters/registers to 0
	GlobalHistoryRegister = 0x0;
	hybridCounter = 0x2;	// Initialized to a weak taken

	/* if using GAg scheme, initialize the HistoryPatternTable
	 *
	 * HINT: in the case of GAg, you will need to add a history register and history table
	 * Something like:
	 *    unsigned int BTB_History = 0;
	 *    unsigned int BTB_HistoryLength = 8;
	 * 	  unsigned char BTB_Table[256];
	 *
	 * The table would be initialized to 2 for weak taken, and you would access the table using:
	 * 	prediction = BTB_Table[BTB_History & 0xFF];
	 * 	if (prediction > 1)
	 * 		prediction is taken
	 * 	else
	 * 		prediction is not taken
	 *
	 * Hybrid scheme: combines 2-bit and GAg
	 */
	if ((predictionType == GAg) | (predictionType == HYBRID)) {
		// initialize values in HistoryPatternTable
		// Dynamically assign size of HistoryPatternTable
		//	UINT8 HistorySize = 8;
		//	HistoryPatternTable = new unsigned char[pow(2.0, HistorySize)];

		// assign space for table with required size (8-bit history, 2^8 entries in PatternTable)
		HistoryPatternTable = new unsigned char[BTB_GAG_TABLE_SIZE];

		// initialize values in table, all initialized to 'Taken'
		for (i = 0; i < BTB_GAG_TABLE_SIZE; i++){
			HistoryPatternTable[i] = 0x2;
		}
	}

	/*
	 * If using PAg scheme, initialize HistoryPatternTable
	 */
	if (predictionType == PAg){
		// initialize values in BTB_Table
		// Dynamically assign size of HistoryPatternTable
		//	UINT8 HistorySize = 8;
		//	HistoryPatternTable = new unsigned char[pow(2.0, HistorySize)];

		// assign space for table with required size (12-bit history, 2^12 entries in PatternTable)
		HistoryPatternTable = new unsigned char[BTB_PAG_TABLE_SIZE];

		// initialize values in table, all initialized to 'Taken'
		for (i = 0; i < BTB_PAG_TABLE_SIZE; i++){
			HistoryPatternTable[i] = 0x2;
		}
	}
}

/* see if the given address is in the BTB */
bool BTB_lookup(ADDRINT ins_ptr)
{
	UINT64 index;
	index = mask & ins_ptr;
	if(BTB[index].valid)
		if(BTB[index].tag == ins_ptr)
			return true;
	return false;
}

/* return the prediction for the given address */
bool BTB_prediction(ADDRINT ins_ptr)
{
	UINT64 index;
	index = mask & ins_ptr;
	int predict;

	switch (predictionType){
	case TWO_BIT:
		// Return based on strength of counter (0-1: Not taken, 2-3: Taken)
		return (BTB[index].counter > 1 ? 1:0);
		break;
	case GAg:
		// Use 8-bit history in HistoryRegister
		return (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] > 1 ? 1:0);
	case PAg:
		// Use 12-bit history in individual branchHistoryRegister based on index
		predict = HistoryPatternTable[(BTB[index].branchHistoryRegister & PAgMASK)];
		return (predict > 1 ? 1:0);
	case HYBRID:
		// if hybridCounter predicts 'Taken,' return the GAg prediction. Otherwise, return 2-bit prediction
		if (hybridCounter > 1) {
			return (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] > 1? 1:0);
		}
		else{
			return (BTB[index].counter > 1 ? 1:0);
		}
	default:
		// Return prediction stored in individual BTB entry
		return BTB[index].prediction;
	}
}

/* update the BTB entry with the last result */
VOID BTB_update(ADDRINT ins_ptr, bool taken)
{
	UINT64 index;
	index = mask & ins_ptr;

	switch (predictionType){
	case TWO_BIT:
		// Update if branch was taken, increase strength of taken prediction
		if (taken) {
			if (BTB[index].counter < 3) BTB[index].counter++;
		}
		// Update if branch was not taken, increase strength of not-taken prediction
		else {
			if (BTB[index].counter > 0) BTB[index].counter--;
		}
		return;
	case GAg:


		// Update counter in HistoryTable
		if (taken) {
			if (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] < 3) {
				HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)]++;
			}
		}
		// Update if branch was not taken, increase strength of not-taken prediction
		else {
			if (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] > 0) {
				HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)]--;
			}
		}
		// Shift in new taken history into the global history register
		GlobalHistoryRegister = (GlobalHistoryRegister << 1) | taken;
		return;
	case PAg:

		// Update counter in HistoryTable based on branchHistoryRegister
		if (taken) {
			if (HistoryPatternTable[(BTB[index].branchHistoryRegister & PAgMASK)] < 3) {
				HistoryPatternTable[(BTB[index].branchHistoryRegister & PAgMASK)]++;
			}
		}
		// Update if branch was not taken, increase strength of not-taken prediction
		else {
			if (HistoryPatternTable[(BTB[index].branchHistoryRegister & PAgMASK)] > 0) {
				HistoryPatternTable[(BTB[index].branchHistoryRegister & PAgMASK)]--;
			}
		}
		// Shift in new taken history into branch history register
		BTB[index].branchHistoryRegister = (BTB[index].branchHistoryRegister << 1) | taken;
		return;
	case HYBRID:
		if (hybridCounter > 1) {
			// Update the GAg scheme


			// Update counter in HistoryTable
			if (taken) {
				if (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] < 3) {
					HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)]++;
				}
			}
			// Update if branch was not taken, increase strength of not-taken prediction
			else {
				if (HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)] > 0) {
					HistoryPatternTable[(GlobalHistoryRegister & GAgMASK)]--;
				}
			}
			// Shift in new taken history into the global history register
			GlobalHistoryRegister = (GlobalHistoryRegister << 1) | taken;
		}
		else {
			// Update the 2-bit scheme
			// Update if branch was taken, increase strength of taken prediction
			if (taken) {
				if (BTB[index].counter < 3) BTB[index].counter++;
			}
			// Update if branch was not taken, increase strength of not-taken prediction
			else {
				if (BTB[index].counter > 0) BTB[index].counter--;
			}
		}
		// Now update the hybrid counter
		if (taken) {
			if (hybridCounter < 3) hybridCounter++;
		}
		else {
			if (hybridCounter > 0) hybridCounter--;
		}
		return;
	default:
		// Set prediction based on what was last done
		BTB[index].prediction = taken;
		break;
	}

}

/* insert a new branch in the table */
VOID BTB_insert(ADDRINT ins_ptr)
{
	UINT64 index;
	index = mask & ins_ptr;

	if(BTB[index].valid)
	{
		BTB[index].ReplaceCount++;
		CountReplaced++;
	}

	BTB[index].valid = true;
	BTB[index].tag = ins_ptr;

	switch (predictionType) {
	case TWO_BIT:
		BTB[index].counter = 2;	 // Predict next branch as weakly taken [2-bit];
		break;
	case GAg:
		// Shift in 'Taken' into the global history register
		GlobalHistoryRegister = (GlobalHistoryRegister << 1) | 0x1;
		break;
	case PAg:
		BTB[index].branchHistoryRegister = 0x1;	// Set up taken in new history
		HistoryPatternTable[BTB[index].branchHistoryRegister && GAgMASK] = 0x2;	// Set history pattern table counter entry to Taken
		break;
	case HYBRID:
		BTB[index].counter = 2;					// Predict next branch as weakly taken [2-bit];
		GlobalHistoryRegister = (GlobalHistoryRegister << 1) | 0x1;
		HistoryPatternTable[GlobalHistoryRegister && GAgMASK] = 0x2;	// Set history pattern table counter entry to Taken
		break;
	default:
		BTB[index].prediction = true;  			// Missed branches always enter as taken/true
	}
}

/* ===================================================================== */


/* ===================================================================== */

VOID WriteResults(bool limit_reached)
{
	int i;

	string output_file = KnobOutputFile.Value();
	if(KnobPid) output_file += "." + decstr(getpid());

	std::ofstream out(output_file.c_str());
	if(limit_reached)
		out << "Reason: limit reached\n";
	else
		out << "Reason: finite\n";

	// Determine branch prediction scheme used
	string predictionTypeString;
	switch (predictionType) {
	case TWO_BIT:
		predictionTypeString = "Two-Bit Prediction";
		break;
	case GAg:
		predictionTypeString = "GAg Prediction";
		break;
	case PAg:
		predictionTypeString = "PAg Prediction";
		break;
	case HYBRID:
		predictionTypeString = "Hybrid Predictor: Combining 2-bit and GAg Predictions";
		break;
	default:
		predictionTypeString = "One-Bit Prediction";
		break;
	}
	out << "Branch Prediction Used: " << predictionTypeString << endl;
	out << "Count Seen: " << CountSeen << endl;
	out << "Count Taken: " << CountTaken << endl;
	out << "Count Correct: " << CountCorrect << endl;
	out << "Count Missed: " << CountMissed << endl;
	out << "Count Replaced: " << CountReplaced << endl;
	out << "Prediction Accuracy: " << (float) CountCorrect / CountSeen << endl;
	for(i = 0; i < BTB_SIZE; i++)
	{
		out << "BTB entry " << i << "\tValid: " << BTB[i].valid << "\tReplace Count: " << BTB[i].ReplaceCount << endl;
	}
	out.close();
}

/* ===================================================================== */

VOID br_predict(ADDRINT ins_ptr, INT32 taken)
{
	CountSeen++;
	if (taken)
		CountTaken++;

	if(BTB_lookup(ins_ptr))
	{
		if(BTB_prediction(ins_ptr) == taken) CountCorrect++;
		BTB_update(ins_ptr, taken);
	}
	else
	{
		CountMissed++;	// Keep track of miss rate
		if(!taken) {
			CountCorrect++;	// Correctly predicted 'Not Taken'
			// Shift in new taken history into the global history register
			if ((predictionType == GAg) | (predictionType == HYBRID)) {
				GlobalHistoryRegister = (GlobalHistoryRegister << 1) | taken;
				HistoryPatternTable[GlobalHistoryRegister && GAgMASK] = 0x1;		// Set History Pattern Table to Not Taken
			}
		}
		else BTB_insert(ins_ptr);
	}

	if(CountSeen == KnobBranchLimit.Value())
	{
		WriteResults(true);
		exit(0);
	}
}


//  IARG_INST_PTR
// ADDRINT ins_ptr

/* ===================================================================== */

VOID Instruction(INS ins, void *v)
{

	// The subcases of direct branch and indirect branch are
	// broken into "call" or "not call".  Call is for a subroutine
	// These are left as subcases in case the programmer wants
	// to extend the statistics to see how sub cases of branches behave

	if( INS_IsRet(ins) )
	{
		INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
				IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
	}
	else if( INS_IsSyscall(ins) )
	{
		INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
				IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
	}
	else if (INS_IsDirectBranchOrCall(ins))
	{
		if( INS_IsCall(ins) ) {
			INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
					IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
		}
		else {
			INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
					IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
		}
	}
	else if( INS_IsIndirectBranchOrCall(ins) )
	{
		if( INS_IsCall(ins) ) {
			INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
					IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
		}
		else {
			INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict,
					IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
		}
	}

}

/* ===================================================================== */

#define OUT(n, a, b) out << n << " " << a << setw(16) << CountSeen. b  << " " << setw(16) << CountTaken. b << endl

VOID Fini(int n, void *v)
{
	WriteResults(false);
}


/* ===================================================================== */


/* ===================================================================== */

int main(int argc, char *argv[])
{

	if( PIN_Init(argc,argv) )
	{
		return Usage();
	}

	// Determine which type of branch prediction scheme is being used
	// Only change default if value is set to 1 (2-bit), 2 (GAg), 3 (PAg) or 4 (Hybrid)
	if(KnobBranchPrediction.Value() > 0 && KnobBranchPrediction.Value() <5 )
	{
		switch (KnobBranchPrediction.Value()){
		// 2-Bit
		case 1:
			predictionType = TWO_BIT;
			break;
		case 2:
			predictionType = GAg;
			break;
		case 3:
			predictionType = PAg;
			break;
		case 4:
			predictionType = HYBRID;
			break;
		}
	}

	BTB_init(); // Initialize hardware structures

	INS_AddInstrumentFunction(Instruction, 0);
	PIN_AddFiniFunction(Fini, 0);

	// Never returns

	PIN_StartProgram();

	return 0;
}

/* ===================================================================== */
/* end of file */
/* ===================================================================== */
