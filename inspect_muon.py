import inspect
import sys
try:
    import muon
    print("Muon package imported successfully.")
    
    if hasattr(muon, 'Muon'):
        print("\n--- Muon.step source code ---")
        try:
            print(inspect.getsource(muon.Muon.step))
        except:
            print("Could not get source for Muon.step")
            
    if hasattr(muon, 'MuonWithAuxAdam'):
        print("\n--- MuonWithAuxAdam.step source code ---")
        try:
            print(inspect.getsource(muon.MuonWithAuxAdam.step))
        except:
            print("Could not get source for MuonWithAuxAdam.step")

    # Check for helper functions that might be involved
    print("\n--- Other functions in muon ---")
    for name, obj in inspect.getmembers(muon):
        if inspect.isfunction(obj) and ('update' in name):
            print(f"\n--- {name} source code ---")
            try:
                print(inspect.getsource(obj))
            except:
                pass

except ImportError:
    print("Could not import muon package.")
