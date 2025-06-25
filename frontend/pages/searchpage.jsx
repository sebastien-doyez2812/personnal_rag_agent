import { useState } from "react";

function SearchPage(){
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    
    // Research function:
    const handleResearch = async (e) => {
        // Avoid a default reload of the page: 
        e.preventDefault();

        // If the query is empty:
        if (!query.trim()){
            setError("Please, answer something to the agent...");
            console.log("[-] No query...");
            return
        }

        // Reset the states:
        setResponse(null);
        setError(null);
        setLoading(true);

        try {
            const res = await fetch("http://127.0.0.1:8000/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: query }),
            });
            
            console.log("Backend response:", res); 

            // Test if agent is ok:
            if (!res.ok){
                const errorData = await res.json();
                try {
                    if (errorData.detail){
                        errorMessage = `Agent Error: ${errorData.detail}`;
                    }
                    else if (errorData.message){
                        errorMessage = `Agent Error: ${errorData.message}`;
                    }
                }
                catch (jsonError) {
                    errorMessage = `Erreur du serveur (Statut HTTP: ${res.status} ${res.statusText}).`;
                }
                finally{
                    throw new Error(errorMessage);
                }
            }
            
            // else get the answer:
            const data = await res.json();
            setResponse(data.response);
        }
        catch(err) {
            console.log(err);
            setError(err);
        }
        finally {
            setLoading(false);
        }
    }

    return <>
    <div className="search_page">
        <h1 className="title_search_page" >Personnal Rag agent for Seb Doyez</h1>
        <form onSubmit={handleResearch} className="search_form" >
            <input type="text" 
                    placeholder="What is deeplearning?"
                    className="input_search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading} // disable the input during a loading phase
            />
            <button type="submit"
                    disabled= {loading || !query.trim()}
                    >
                {loading? "Fetching data... " : "Ask me a question!"}
            </button>
        </form>
        {
            error && (
                <div className="error_container">     <p className="font-bold">Erreur :</p>
                <p>{error.message || error.toString()}</p>
                </div>
            )
        }
        {
            response && (
                <div className="answer_container">
                    <h2 className="answer">
                        Answer:
                    </h2>
                    <p className="agent_answer">
                        {response}
                    </p>
                </div>

            )
        }
    </div>
    </>
}

export default SearchPage;