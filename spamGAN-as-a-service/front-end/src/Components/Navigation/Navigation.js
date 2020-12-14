import React from 'react';
import '../../css/default.css';
import '../../css/layout.css';
import '../../css/media-queries.css';

const navigation = (props) => (
    <header id="home" className='navigation'>
        <nav id="nav-wrap">
            <a className="mobile-btn" href="#nav-wrap" title="Show navigation">Show navigation</a>
            <a className="mobile-btn" href="#home" title="Hide navigation">Hide navigation</a>

            <ul id="nav" className="nav">
                <li className="current"><a className="smoothscroll" href="#home">Home</a></li>
                <li><a className="smoothscroll" href="#test">Inference</a></li>
            </ul>
        </nav>

        <div className="row banner">
      
            <div className="banner-text">
                <h1 className="responsive-headline" >Review Spam Detection</h1>
                <h3>Welcome to Review Spam Detection website. You can input review or upload a file to see if it is spam or non-spam.</h3>
                <hr />
                <ul className="social">
                <a  className="button btn project-btn" target = "_blank"><i className="fa fa-book"></i>Paper</a>
                <a  className="button btn github-btn" href='https://github.com/airesearchuwt/spamGAN' target="_blank"><i className="fa fa-github"></i>Github</a>
                </ul>
            </div>
        </div>

      <p className="scrolldown">
         <a className="smoothscroll" href="#test"><i className="icon-down-circle"></i></a>
      </p>
        
    </header>
);

export default navigation;